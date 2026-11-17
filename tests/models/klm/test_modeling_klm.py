# Copyright 2026 The KLM authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch KLM model."""

import contextlib
import copy
import json
import math
import re
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parameterized import parameterized

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, KlmConfig, is_torch_available
from transformers.testing_utils import cleanup, require_accelerate, require_torch, slow, torch_device
from transformers.utils import is_kernels_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    from transformers import AutoTokenizer, DynamicCache, KlmForCausalLM, KlmModel
    from transformers.models.klm import modeling_klm as klm
    from transformers.models.klm.convert_klm_weights_to_hf import convert as convert_checkpoint
    from transformers.models.klm.convert_klm_weights_to_hf import convert_config, convert_state_dict
    from transformers.models.klm.modeling_klm import (
        KlmRotaryEmbedding,
        torch_chunk_gated_delta_rule_2,
        torch_recurrent_gated_delta_rule_2,
    )


class KlmModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = KlmConfig
        base_model_class = KlmModel
        causal_lm_class = KlmForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_num_heads = 2
        self.linear_head_dim = 8
        self.linear_conv_kernel_dim = 4
        self.num_bigrams = 24
        self.num_trigrams = 16
        self.ngram_embedding_size = 8
        self.hidden_act = "silu"


@require_torch
class KlmModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = KlmModelTester
    rotary_embedding_layer = KlmRotaryEmbedding if is_torch_available() else None

    def _get_model(self, config=None):
        if config is None:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.eos_token_id = 2
        model = KlmModel(config).to(torch_device).eval()
        # Every block residual starts as a plain mean, random queries make the test sensitive to the order of states
        for name, parameter in model.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter)
        return model

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        """Overwritten: only the first linear attention layer uses its second conv state, for the document state."""
        self.assertIsInstance(cache, DynamicCache)
        self.assertEqual(len(cache), config.num_hidden_layers)
        qkv_dim = config.linear_num_heads * config.linear_head_dim
        k_shape, v_shape = self._get_attention_shape(batch_size, seq_length, config)
        document_layer_idx = config.layer_types.index("linear_attention")
        for layer_idx, layer in enumerate(cache.layers):
            if config.layer_types[layer_idx] == "full_attention":
                self._check_attention_shapes(layer, seq_length, k_shape, v_shape)
                continue
            self.assertEqual(layer.conv_states[0].shape, (batch_size, 3 * qkv_dim, config.linear_conv_kernel_dim))
            self.assertEqual(
                layer.recurrent_states[0].shape,
                (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim),
            )
            if layer_idx == document_layer_idx:
                self.assertEqual(layer.conv_states[1].shape, (batch_size, 3, 1))
            else:
                self.assertIsNone(layer.conv_states[1])

    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip("Layers read all the previous hidden states, which only non-reentrant checkpointing can pass")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def test_attention_outputs(self):
        """Needs to be overwritten as KLM alternates between attention layers and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_chunk_gated_delta_rule_matches_recurrent(self):
        """The chunked kernel has to give the outputs and final state of the token by token one."""
        batch_size, seq_len, num_heads, head_dim = 3, 75, 2, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        query, key, value = torch.randn(3, *shape, dtype=torch.float64, device=torch_device)
        decay = -3 * torch.rand(shape, dtype=torch.float64, device=torch_device)
        erase, write = torch.rand(2, *shape, dtype=torch.float64, device=torch_device)
        # Documents that start inside a chunk, one right after another, and on a chunk boundary
        document_starts = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=torch_device)
        document_starts[0, [9, 10, 40]] = True
        document_starts[1, [32, 64]] = True
        initial_state = torch.randn(batch_size, num_heads, head_dim, head_dim, device=torch_device)

        for state in (None, initial_state):
            inputs = (query, key, value, decay, erase, write, document_starts, state)
            expected_output, expected_state = torch_recurrent_gated_delta_rule_2(*inputs)
            for chunk_size in (1, 7, 32, 128):
                output, final_state = torch_chunk_gated_delta_rule_2(*inputs, chunk_size=chunk_size)
                torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(final_state, expected_state, rtol=1e-4, atol=1e-5)

    def test_documents_are_independent(self):
        """Whatever follows an end of sequence token is read as if nothing came before it."""
        model = self._get_model()
        vocab_size, eos_token_id = model.config.vocab_size, model.config.eos_token_id
        first = ids_tensor((2, 9), vocab_size - 3).to(torch_device) + 3
        second = ids_tensor((2, 40), vocab_size - 3).to(torch_device) + 3
        eos = torch.full((2, 1), eos_token_id, device=torch_device)

        for implementation in ("eager", "sdpa"):
            model.set_attn_implementation(implementation)
            with torch.no_grad():
                expected = model(second).last_hidden_state
                output = model(torch.cat([first, eos, second], dim=1)).last_hidden_state[:, -second.shape[1] :]
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_left_padding_is_not_read(self):
        """The first token after left padding starts a document, so the padding cannot change the outputs."""
        model = self._get_model()
        input_ids = ids_tensor((2, 20), model.config.vocab_size - 3).to(torch_device) + 3
        padding = torch.full((2, 6), model.config.pad_token_id, device=torch_device)
        attention_mask = torch.cat([torch.zeros_like(padding), torch.ones_like(input_ids)], dim=1)

        with torch.no_grad():
            expected = model(input_ids).last_hidden_state
            output = model(torch.cat([padding, input_ids], dim=1), attention_mask=attention_mask).last_hidden_state
        torch.testing.assert_close(output[:, padding.shape[1] :], expected, rtol=1e-4, atol=1e-4)

    def test_cached_forward_matches_full_forward(self):
        """
        Feeding a sequence in pieces with a cache has to give the outputs of a single forward, whether a piece is one
        token or several, and whether documents end inside a piece or between two of them.
        """
        model = self._get_model()
        input_ids = ids_tensor((2, 60), model.config.vocab_size - 3).to(torch_device) + 3
        input_ids[0, [4, 5, 33]] = model.config.eos_token_id
        input_ids[1, [17, 41]] = model.config.eos_token_id

        with torch.no_grad():
            expected = model(input_ids).last_hidden_state
            for pieces in ([1] * 60, [18, 1, 1, 22, 1, 17], [5, 55]):
                cache = DynamicCache(config=model.config)
                outputs = [
                    model(piece, cache=cache, use_cache=True).last_hidden_state
                    for piece in input_ids.split(pieces, dim=1)
                ]
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-4, atol=1e-4)

    def test_autocast(self):
        """The recurrence of the linear attention layers stays in float32 under autocast."""
        torch.manual_seed(0)
        model = self._get_model(self.model_tester.get_config())
        input_ids = torch.arange(80, device=torch_device).reshape(2, 40) % model.config.vocab_size

        def check_recurrence(*args, **kwargs):
            self.assertFalse(torch.is_autocast_enabled(torch_device))
            output, state = torch_chunk_gated_delta_rule_2(*args, **kwargs)
            self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(state.dtype, torch.float32)
            # Compare the same projected operands, so this checks recurrence precision independently of BF16
            # projection rounding and its amplification by the randomly initialized residual routing.
            expected_output, expected_state = torch_recurrent_gated_delta_rule_2(*args, **kwargs)
            torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-5)
            return output, state

        with patch(
            "transformers.models.klm.modeling_klm.torch_chunk_gated_delta_rule_2", side_effect=check_recurrence
        ) as recurrence:
            with torch.no_grad(), torch.autocast(torch_device, dtype=torch.bfloat16):
                output = model(input_ids).last_hidden_state
        self.assertEqual(recurrence.call_count, model.config.layer_types.count("linear_attention"))
        self.assertTrue(torch.isfinite(output).all())

    @parameterized.expand([("single_token", 1), ("shorter_padding", 2), ("longer_padding", 3), ("inside_document", 4)])
    def test_cached_left_padding(self, _, chunk_size):
        model = self._get_model()
        input_ids = torch.tensor([[0, 0, 0, 5, 6, 7], [0, 0, 3, 8, 9, 4]], device=torch_device)
        attention_mask = input_ids.ne(0)
        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            cache = DynamicCache(config=model.config)
            outputs = []
            for start in range(0, input_ids.shape[1], chunk_size):
                end = min(start + chunk_size, input_ids.shape[1])
                outputs.append(
                    model(
                        input_ids[:, start:end], attention_mask=attention_mask[:, :end], cache=cache, use_cache=True
                    ).last_hidden_state
                )
        actual = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual[attention_mask], expected[attention_mask], rtol=1e-4, atol=1e-4)

    def test_generate_chunked_prefill_left_padding(self):
        torch.manual_seed(41)
        config = self.model_tester.get_config()
        model = KlmForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[0, 0, 0, 5, 6, 7], [0, 0, 3, 8, 9, 4]], device=torch_device)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
            "max_new_tokens": 3,
            "do_sample": False,
            "eos_token_id": None,
            "pad_token_id": 0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        with torch.no_grad():
            expected = model.generate(**kwargs)
            actual = model.generate(**kwargs, prefill_chunk_size=3)
        torch.testing.assert_close(actual.sequences, expected.sequences)
        for actual_scores, expected_scores in zip(actual.scores, expected.scores):
            torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_ngram_embeddings(self):
        """N-grams are looked up by key and do not reach back past the start of a document."""
        model = self._get_model()
        embed_ngrams, vocab_size = model.embed_ngrams, model.config.vocab_size
        input_ids = torch.tensor([[5, 6, 7, 2, 5, 6, 7, 8]], device=torch_device)
        document_starts = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.bool, device=torch_device)
        previous_ids = torch.zeros((1, 2), dtype=torch.long, device=torch_device)
        # Known n-grams: (5, 6), (7, 8) and (5, 6, 7). The bank is sorted, the unused keys are larger than any n-gram.
        bigram_keys = torch.full_like(embed_ngrams.bigram_keys, vocab_size**3)
        bigram_keys[:2] = torch.tensor([5 * vocab_size + 6, 7 * vocab_size + 8])
        trigram_keys = torch.full_like(embed_ngrams.trigram_keys, vocab_size**3)
        trigram_keys[0] = (5 * vocab_size + 6) * vocab_size + 7
        embed_ngrams.bigram_keys.copy_(bigram_keys)
        embed_ngrams.trigram_keys.copy_(trigram_keys)

        with torch.no_grad():
            output = embed_ngrams(input_ids, previous_ids, document_starts)
            unknown = embed_ngrams.bigram_proj(embed_ngrams.bigram_embeddings.weight[0])
            unknown = unknown + embed_ngrams.trigram_proj(embed_ngrams.trigram_embeddings.weight[0])
            bigram = embed_ngrams.bigram_proj(embed_ngrams.bigram_embeddings.weight[1:3])
            trigram = embed_ngrams.trigram_proj(embed_ngrams.trigram_embeddings.weight[:2])
        # Positions 0 and 4 have no previous token, position 3 is not a known n-gram
        for position in (0, 3, 4):
            torch.testing.assert_close(output[0, position], unknown)
        # (5, 6) is known at positions 1 and 5, both times without a known trigram
        torch.testing.assert_close(output[0, 1], bigram[0] + trigram[0])
        torch.testing.assert_close(output[0, 5], output[0, 1])
        # (5, 6, 7) is known at positions 2 and 6, where (6, 7) is not a known bigram
        torch.testing.assert_close(output[0, 2], unknown - trigram[0] + trigram[1])
        torch.testing.assert_close(output[0, 6], output[0, 2])
        # (7, 8) is known at position 7, where (6, 7, 8) is not a known trigram
        torch.testing.assert_close(output[0, 7], bigram[1] + trigram[0])


@require_torch
class KlmIntegrationTest(unittest.TestCase):
    # The original checkpoint, converted with `convert_klm_weights_to_hf.py`
    checkpoint = "MinimaLabs/KLM-1.6"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = KlmForCausalLM.from_pretrained(self.checkpoint).to(torch_device)
        # Two documents in a single sequence
        text = "The capital of France is Paris.<eos>Photosynthesis is the process by which plants"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(torch_device)
        with torch.no_grad():
            logits = model(input_ids).logits.float()

        EXPECTED_MEAN = torch.tensor(
            [[18.7292, -2.1244, 4.1339, -6.1907, 3.4924, 1.3876, 0.4001, -0.5738]], device=torch_device
        )
        torch.testing.assert_close(logits.mean(-1)[:, -8:], EXPECTED_MEAN, rtol=1e-3, atol=1e-3)
        EXPECTED_SLICE = torch.tensor(
            [-7.6502, -7.7844, 6.0391, 2.5945, 4.9552, -1.9173, -5.5932, -1.0267, 3.2742, 7.5289], device=torch_device
        )
        torch.testing.assert_close(logits[0, -1, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-3)

    @slow
    def test_model_greedy_generation(self):
        EXPECTED_TEXT_COMPLETIONS = [
            "Photosynthesis is the process by which plants turn light into chemical energy. The process is carried out by the chloroplasts of the plant cells. The chloropl",
            "Once upon a time there was a small fox who lived at the edge of a forest. The forest was full of tall trees and bushes, but the fox loved to hide in the",
        ]
        prompts = [
            "Photosynthesis is the process by which plants turn light into chemical energy.",
            "Once upon a time there was a small fox who lived at the edge of a forest.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, padding_side="left")
        model = KlmForCausalLM.from_pretrained(self.checkpoint).to(torch_device)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETIONS, text)


@require_torch
class KlmCheckpointConversionTest(unittest.TestCase):
    def test_default_checkpoint_reloads_directly_without_replacing_parameters(self):
        # Unsloth DecisionTrainer uses encoder.load_state_dict(), not from_pretrained(),
        # when restoring its best checkpoint or resuming a full fine-tune.
        with tempfile.TemporaryDirectory() as source:
            self._original_checkpoint(source)
            for auto_class in (AutoModel, AutoModelForCausalLM):
                with self.subTest(auto_class=auto_class.__name__):
                    model = auto_class.from_pretrained(source)
                    optimizer = torch.optim.AdamW(model.parameters())
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(0.125)
                    expected = {key: value.clone() for key, value in model.state_dict().items()}
                    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
                    with tempfile.TemporaryDirectory() as saved:
                        model.save_pretrained(saved)
                        state = load_file(str(Path(saved) / "model.safetensors"))
                        self.assertTrue(any("attn_res_query" in key for key in state))
                        keys = list(state)
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.zero_()
                        missing, unexpected = model.load_state_dict(state, strict=False)
                        self.assertEqual(missing, [])
                        self.assertEqual(unexpected, [])
                        self.assertEqual(list(state), keys)
                        model.load_state_dict(state, strict=True)
                    self.assertEqual(
                        {name: id(parameter) for name, parameter in model.named_parameters()}, parameter_ids
                    )
                    self.assertEqual(
                        {id(p) for group in optimizer.param_groups for p in group["params"]},
                        set(parameter_ids.values()),
                    )
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_partial_original_projection_and_assign(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
        parameter = model.model.layers[0].linear_attn.b_proj.weight
        value = torch.randn_like(parameter, dtype=torch.float64)
        state = {"model.blocks.0.mixer.b_proj.weight": value}
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        self.assertNotIn("model.layers.0.linear_attn.b_proj.weight", missing)
        self.assertEqual(unexpected, [])
        self.assertEqual(model.model.layers[0].linear_attn.b_proj.weight.dtype, torch.float64)
        torch.testing.assert_close(model.model.layers[0].linear_attn.b_proj.weight, value, rtol=0, atol=0)
        self.assertEqual(list(state), ["model.blocks.0.mixer.b_proj.weight"])

    def test_mixed_original_and_native_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        original["model.layers.0.input_residual.query"] = model.model.layers[0].input_residual.query.detach().clone()
        with self.assertRaisesRegex(ValueError, "original and native"):
            model.load_state_dict(original, strict=False)

    def test_native_partial_state_keeps_pytorch_strictness(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
        state = {"model.norm.weight": model.model.norm.weight.detach().clone(), "unknown.weight": torch.zeros(1)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertTrue(missing)
        self.assertEqual(unexpected, ["unknown.weight"])
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state)

    def test_untied_output_weight_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        model.config.tie_word_embeddings = False
        model.lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
        expected = model.lm_head.weight.detach().clone()
        missing, unexpected = model.load_state_dict(original, strict=False)
        self.assertEqual(missing, ["lm_head.weight"])
        self.assertEqual(unexpected, [])
        torch.testing.assert_close(model.lm_head.weight, expected, rtol=0, atol=0)

    def test_converter_writes_native_checkpoint(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            self._original_checkpoint(source)
            # Tokenizer copying is independent of the weight layout under test.
            with patch("transformers.models.klm.convert_klm_weights_to_hf.AutoTokenizer.from_pretrained"):
                convert_checkpoint(source, target)
            state = load_file(str(Path(target) / "model.safetensors"))
            self.assertIn("model.layers.0.linear_attn.conv1d.weight", state)
            self.assertNotIn("model.attn_res_query", state)
            restored = AutoModelForCausalLM.from_pretrained(target)
            missing, unexpected = restored.load_state_dict(state, strict=False)
            self.assertLessEqual(set(missing), {"lm_head.weight"})
            self.assertEqual(unexpected, [])

    def _original_checkpoint(self, directory):
        config = {
            "model_type": "klm",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "attention_every": 2,
            "gdn_heads": 2,
            "gdn_head_dim": 8,
            "gdn_norm_eps": 2e-5,
            "decay_floor": -12.0,
            "n_bigrams": 7,
            "n_trigrams": 5,
            "ngram_dim": 4,
            "rope_theta": 12345.0,
            "tie_word_embeddings": True,
        }
        model = KlmForCausalLM(convert_config(config)).eval()
        original = {}
        residuals = {"query": [], "weight": []}
        for name, tensor in model.state_dict().items():
            if name == "lm_head.weight":
                continue
            if name.endswith(("_residual.query", "_residual.weight")):
                residuals[name.rsplit(".", 1)[1]].append(torch.randn_like(tensor))
                continue
            name = name.replace("model.embed_tokens.", "model.embed.")
            name = name.replace("model.embed_ngrams.", "model.memory.")
            name = name.replace("bigram_embeddings.", "bigram.").replace("trigram_embeddings.", "trigram.")
            name = name.replace("model.layers.", "model.blocks.")
            name = name.replace(".input_layernorm.", ".mixer_norm.")
            name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
            name = re.sub(r"\.mlp\.(gate|up|down)_proj\.", r".ffn.\1.", name)
            name = name.replace(".self_attn.", ".mixer.").replace(".linear_attn.", ".mixer.")
            name = re.sub(r"\.(f|g)_a_proj\.", r".\1_proj.0.", name)
            name = re.sub(r"\.(f|g)_b_proj\.", r".\1_proj.1.", name)
            if name.endswith(".mixer.conv1d.weight"):
                for component, weight in zip("qkv", tensor.chunk(3, dim=0)):
                    original[name.replace("conv1d", f"{component}_conv1d")] = weight.clone()
            else:
                original[name] = tensor.clone()
        original["model.attn_res_query"] = torch.stack(residuals["query"])
        original["model.attn_res_norm"] = torch.stack(residuals["weight"])
        Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(original, Path(directory, "model.safetensors"))
        model.load_state_dict(convert_state_dict(original, model.config), strict=False)
        return model

    def test_native_loading_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self._original_checkpoint(directory)
            config = AutoConfig.from_pretrained(directory)
            self.assertEqual(config.linear_num_heads, 2)
            self.assertEqual(config.linear_head_dim, 8)
            self.assertEqual(config.num_bigrams, 7)
            self.assertEqual(config.layer_types, ["linear_attention", "full_attention"] * 2)
            self.assertEqual(config.rope_parameters["rope_theta"], 12345.0)
            input_ids = torch.tensor([[1, 4, 9, 2, 6, 7]])
            for auto_class, expected in ((AutoModel, reference.model), (AutoModelForCausalLM, reference)):
                with self.subTest(auto_class=auto_class.__name__):
                    loaded, info = auto_class.from_pretrained(directory, output_loading_info=True)
                    self.assertFalse(info["missing_keys"])
                    self.assertFalse(info["unexpected_keys"])
                    for key, tensor in expected.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(input_ids)[0], expected(input_ids)[0])
                    for save_original_format in (True, False):
                        with tempfile.TemporaryDirectory() as saved:
                            loaded.save_pretrained(saved, save_original_format=save_original_format)
                            restored, info = auto_class.from_pretrained(saved, output_loading_info=True)
                            self.assertFalse(info["missing_keys"])
                            self.assertFalse(info["unexpected_keys"])
                            for key, tensor in loaded.state_dict().items():
                                torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)

    def test_what_cannot_be_converted_is_reported(self):
        # As PyTorch does with any name it does not know: reported without `strict`, an error with it
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        rows = original["model.attn_res_query"]
        for name, state in (
            (
                "model.blocks.0.mixer.q_conv1d.weight",
                {"model.blocks.0.mixer.q_conv1d.weight": original["model.blocks.0.mixer.q_conv1d.weight"]},
            ),
            ("model.attn_res_query", {"model.attn_res_query": rows[:-1]}),
            ("model.blocks.9.mixer.o_proj.weight", {"model.blocks.9.mixer.o_proj.weight": torch.zeros(1)}),
        ):
            with self.subTest(name=name):
                state["model.norm.weight"] = torch.full_like(model.model.norm.weight, 3.0)
                missing, unexpected = model.load_state_dict(state, strict=False)
                self.assertEqual(unexpected, [name])
                self.assertIn("model.embed_tokens.weight", missing)
                # The rest is loaded
                torch.testing.assert_close(model.model.norm.weight, state["model.norm.weight"], rtol=0, atol=0)
                with self.assertRaises(RuntimeError):
                    model.load_state_dict(state)

    def test_partial_original_residual_matrix(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        rows = original["model.attn_res_query"] + 1
        missing, unexpected = model.load_state_dict({"model.attn_res_query": rows}, strict=False)
        self.assertEqual(unexpected, [])
        self.assertNotIn("model.layers.0.input_residual.query", missing)
        torch.testing.assert_close(model.model.layers[0].input_residual.query, rows[0], rtol=0, atol=0)

    def test_prefix_of_the_other_class(self):
        # A state dict of the causal language model in the base model, and the other way around
        with tempfile.TemporaryDirectory() as source:
            reference = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        bare = {key.removeprefix("model."): value for key, value in original.items()}
        base_model = KlmModel(reference.config)
        missing, unexpected = base_model.load_state_dict(original, strict=False)
        self.assertEqual((missing, unexpected), ([], []))
        causal_lm = KlmForCausalLM(reference.config)
        missing, unexpected = causal_lm.load_state_dict(bare, strict=False)
        self.assertEqual((missing, unexpected), ([], []))
        self.assertEqual(list(bare), [key.removeprefix("model.") for key in original])
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(causal_lm.state_dict()[name], value, rtol=0, atol=0)
            if name != "lm_head.weight":
                torch.testing.assert_close(base_model.state_dict()[name.removeprefix("model.")], value, rtol=0, atol=0)


@require_torch
@require_accelerate
class KlmOffloadTest(unittest.TestCase):
    def test_offloaded_normalization_hooks_are_preserved(self):
        from accelerate import cpu_offload

        torch.manual_seed(41)
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=8,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=4,
            layer_types=["linear_attention", "full_attention"],
            use_cache=False,
        )
        model = KlmModel(config).eval()
        for name, parameter in model.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        input_ids = torch.tensor([[0, 0, 3, 4, 2, 5, 6], [3, 2, 4, 5, 2, 6, 7]])
        attention_mask = input_ids.ne(0)
        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask).last_hidden_state

        # CPU is the execution device, but weights still become meta tensors between calls. This exercises
        # the same materialization hooks as GPU offload and catches passing a norm module through tensor kwargs.
        cpu_offload(model, execution_device=torch.device("cpu"))
        norms = [model.layers[0].input_layernorm, model.layers[0].post_attention_layernorm, model.norm]
        for _ in range(2):
            for norm in norms:
                self.assertTrue(norm.weight.is_meta)
            with torch.no_grad():
                actual = model(input_ids, attention_mask=attention_mask).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for norm in norms:
                self.assertTrue(norm.weight.is_meta)


@require_torch
class KlmKernelEligibilityTest(unittest.TestCase):
    def test_offload_and_unsupported_widths_use_reference_operators(self):
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=16,
            num_bigrams=8,
            num_trigrams=8,
            ngram_embedding_size=4,
            layer_types=["linear_attention", "full_attention"],
        )
        model = KlmModel(config)
        model._use_fla_kernels = True
        cuda_input = SimpleNamespace(device=torch.device("cuda"))
        self.assertTrue(model._can_use_fla(cuda_input, None))
        model.norm._hf_hook = object()
        self.assertFalse(model._can_use_fla(cuda_input, None))
        del model.norm._hf_hook
        for width in (4, 257):
            model.config.linear_head_dim = width
            self.assertFalse(model._can_use_fla(cuda_input, None))


@require_torch
class KlmAcceleratedIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=16,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=8,
            layer_types=["linear_attention", "full_attention"] * 2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        self.reference = KlmModel(config).train()
        for name, parameter in self.reference.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        self.reference.embed_ngrams.bigram_keys.copy_(torch.arange(16) + 32)
        self.reference.embed_ngrams.trigram_keys.copy_(torch.arange(8) + 32**2 + 32 * 5)
        self.reference._use_fla_kernels = False
        self.accelerated = copy.deepcopy(self.reference)
        self.accelerated._use_fla_kernels = True
        # Construct with the installed CPU backend, then replace FlashAttention with an independent segmented SDPA.
        self.accelerated.config._attn_implementation = "flash_attention_2"
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(
            patch.object(
                KlmModel,
                "_can_use_fla",
                lambda model, ids, cache: model._use_fla_kernels and cache is None,
            )
        )
        self.context.enter_context(
            patch.object(
                KlmModel,
                "_can_use_flash_attention",
                lambda model, ids: model.config._attn_implementation == "flash_attention_2",
            )
        )
        self.attnres = self.context.enter_context(
            patch.object(klm, "klm_fused_attnres", side_effect=self.attnres_reference)
        )
        self.context.enter_context(patch.object(klm, "klm_causal_conv1d", side_effect=self.conv_reference))
        self.gdn = self.context.enter_context(patch.object(klm, "klm_chunk_gdn2", side_effect=self.gdn_reference))
        self.flash = self.context.enter_context(
            patch.dict(
                klm.ALL_ATTENTION_FUNCTIONS._local_mapping,
                {"flash_attention_2": self.flash_reference},
            )
        )

    @staticmethod
    def attnres_reference(query, residuals, rms_weight, output_rms_weight=None, rms_eps=1e-6, **kwargs):
        states = torch.stack(residuals, dim=-2).float()
        normalized = F.rms_norm(states, (states.shape[-1],), eps=rms_eps)
        scores = (normalized * (query * rms_weight).float()).sum(-1)
        output = (scores.softmax(-1).unsqueeze(-1) * states).sum(-2)
        if output_rms_weight is not None:
            output = F.rms_norm(output, (output.shape[-1],), output_rms_weight.float(), eps=rms_eps)
        return output.to(residuals[0].dtype)

    @staticmethod
    def conv_reference(x, weight, activation=None, cu_seqlens=None, cu_seqlens_cpu=None, **kwargs):
        flat = x.reshape(-1, x.shape[-1])
        offsets = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
        outputs = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            part = flat[start:end].T.unsqueeze(0)
            result = F.conv1d(part, weight[:, None], padding=weight.shape[-1] - 1, groups=flat.shape[-1])
            outputs.append(result[..., : end - start].squeeze(0).T)
        output = torch.cat(outputs).reshape_as(x)
        return (F.silu(output) if activation == "silu" else output), None

    @staticmethod
    def gdn_reference(q, k, v, g, b, w, cu_seqlens=None, cu_seqlens_cpu=None, scale=None, **kwargs):
        offsets = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        starts = torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
        starts.reshape(-1)[offsets[:-1].long()] = True
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        outputs = []
        for index in range(q.shape[1]):
            state = state * g[:, index].exp().unsqueeze(-1)
            state = state.masked_fill(starts[:, index, None, None, None], 0)
            erased = ((b[:, index] * k[:, index]).unsqueeze(-1) * state).sum(-2)
            state = state + k[:, index].unsqueeze(-1) * (w[:, index] * v[:, index] - erased).unsqueeze(-2)
            outputs.append((q[:, index].unsqueeze(-1) * state).sum(-2) * (scale or q.shape[-1] ** -0.5))
        return torch.stack(outputs, dim=1), None

    @staticmethod
    def flash_reference(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        offsets = kwargs["cu_seq_lens_q"].tolist()
        length = query.shape[-2]
        mask = torch.zeros((length, length), dtype=torch.bool, device=query.device)
        for start, end in zip(offsets[:-1], offsets[1:]):
            mask[start:end, start:end] = torch.ones((end - start, end - start), dtype=torch.bool).tril()
        groups = query.shape[1] // key.shape[1]
        result = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, 1),
            value.repeat_interleave(groups, 1),
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return result.transpose(1, 2), None

    def inputs(self, kind="padding"):
        ids = torch.tensor([[0, 0, 1, 5, 6, 2, 2, 7, 8], [1, 5, 2, 7, 8, 9, 0, 0, 0]])
        mask = ids != 0
        if kind == "empty_row":
            ids[0] = 0
            mask[0] = False
        elif kind == "holes":
            ids[1, 3] = 0
            mask[1, 3] = False
        elif kind == "empty":
            ids.zero_()
            mask.zero_()
        return ids, mask

    def assert_gradients_match(self):
        actual = dict(self.accelerated.named_parameters())
        for name, expected in self.reference.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(expected.grad)
                self.assertIsNotNone(actual[name].grad)
                torch.testing.assert_close(actual[name].grad, expected.grad, rtol=3e-4, atol=3e-6)

    def test_packed_outputs_and_all_gradients(self):
        for kind in ("padding", "empty_row", "holes"):
            with self.subTest(kind=kind):
                self.reference.zero_grad(set_to_none=True)
                self.accelerated.zero_grad(set_to_none=True)
                ids, mask = self.inputs(kind)
                direction = torch.randn(*ids.shape, 32)
                expected = self.reference(ids, attention_mask=mask).last_hidden_state
                actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(actual[mask], expected[mask], rtol=3e-4, atol=3e-6)
                (expected[mask] * direction[mask]).mean().backward()
                (actual[mask] * direction[mask]).mean().backward()
                self.assert_gradients_match()
        self.assertGreater(self.attnres.call_count, 0)

    def test_checkpoint_multiple_forwards_before_backward(self):
        self.accelerated.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        expected_loss = actual_loss = 0
        for kind in ("padding", "holes"):
            ids, mask = self.inputs(kind)
            direction = torch.randn(*ids.shape, 32)
            expected = self.reference(ids, attention_mask=mask).last_hidden_state
            actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
            expected_loss += (expected[mask] * direction[mask]).mean()
            actual_loss += (actual[mask] * direction[mask]).mean()
        expected_loss.backward()
        actual_loss.backward()
        self.assert_gradients_match()

    def test_causal_lm_retains_padding_queries_that_predict_supervised_tokens(self):
        reference = KlmForCausalLM(copy.deepcopy(self.reference.config)).train()
        reference.model.load_state_dict(self.reference.state_dict())
        reference.model._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        accelerated.model._use_fla_kernels = True
        accelerated.config._attn_implementation = "flash_attention_2"
        ids, mask = self.inputs()
        labels = ids.masked_fill(~mask, -100)
        expected = reference(ids, attention_mask=mask, labels=labels)
        actual = accelerated(ids, attention_mask=mask, labels=labels)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
        actual.loss.backward()
        expected.loss.backward()
        self.reference, self.accelerated = reference, accelerated
        self.assert_gradients_match()
        self.assertEqual(self.gdn.call_args.args[0].shape[1], ids.numel())

    def test_hidden_state_capture_preserves_padded_shapes(self):
        ids, mask = self.inputs()
        expected = self.reference(ids, attention_mask=mask, output_hidden_states=True)
        actual = self.accelerated(ids, attention_mask=mask, output_hidden_states=True)
        self.assertEqual(self.attnres.call_count, 0)
        self.assertEqual(len(actual.hidden_states), len(expected.hidden_states))
        for x, y in zip(actual.hidden_states, expected.hidden_states):
            self.assertEqual(x.shape, (*ids.shape, 32))
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)

    def test_external_causal_loss_retains_padding_position_logits(self):
        reference = KlmForCausalLM(copy.deepcopy(self.reference.config)).train()
        reference.model.load_state_dict(self.reference.state_dict())
        reference.model._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        accelerated.model._use_fla_kernels = True
        accelerated.config._attn_implementation = "flash_attention_2"
        ids, mask = self.inputs()
        labels = ids.masked_fill(~mask, -100)
        expected = reference(ids, attention_mask=mask).logits
        actual = accelerated(ids, attention_mask=mask).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for logits in (expected, actual):
            F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)).backward()
        self.reference, self.accelerated = reference, accelerated
        self.assert_gradients_match()
        self.assertEqual(self.gdn.call_args.args[0].shape[1], ids.numel())

    def test_cache_never_uses_training_kernels(self):
        ids, mask = self.inputs()
        expected_cache, actual_cache = (
            DynamicCache(config=self.reference.config),
            DynamicCache(config=self.accelerated.config),
        )
        for start, end in ((0, 4), (4, 9)):
            expected = self.reference(
                ids[:, start:end], attention_mask=mask[:, :end], cache=expected_cache, use_cache=True
            )
            actual = self.accelerated(
                ids[:, start:end], attention_mask=mask[:, :end], cache=actual_cache, use_cache=True
            )
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
        self.assertEqual(self.attnres.call_count, 0)

    def test_attention_capture_uses_portable_eager_path(self):
        ids, mask = self.inputs()
        self.reference.config._attn_implementation = "eager"
        expected = self.reference(ids, attention_mask=mask, output_attentions=True)
        actual = self.accelerated(ids, attention_mask=mask, output_attentions=True)
        self.assertEqual(self.attnres.call_count, 0)
        self.assertEqual(len(actual.attentions), len(expected.attentions))
        for x, y in zip(actual.attentions, expected.attentions):
            self.assertEqual(x.shape, (ids.shape[0], 4, ids.shape[1], ids.shape[1]))
            torch.testing.assert_close(x, y)

    def test_all_masked_input_falls_back(self):
        ids, mask = self.inputs("empty")
        expected = self.reference(ids, attention_mask=mask).last_hidden_state
        actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.attnres.call_count, 0)


CUDA_FLA = is_torch_available() and torch.cuda.is_available() and klm.is_klm_fla_available()
SMOKE_METRICS = []
FLASH_REVISION = "291664546de978ab2bb730554252e5e2c1099e4f"


@unittest.skipUnless(CUDA_FLA, "Requires CUDA and the optional FLA KLM kernels")
class KlmAcceleratedCudaTest(unittest.TestCase):
    @staticmethod
    def _error(actual, reference):
        squared_error = squared_norm = maximum = 0.0
        for a, b in zip(actual, reference, strict=True):
            a, b = a.detach().double().cpu(), b.detach().double().cpu()
            difference = a - b
            squared_error += difference.square().sum().item()
            squared_norm += b.square().sum().item()
            maximum = max(maximum, difference.abs().max().item())
        return {"relative_l2": math.sqrt(squared_error / max(squared_norm, 1e-30)), "max_abs": maximum}

    def _setup_case(self, amp, causal_lm=False):
        torch.manual_seed(3407)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            linear_num_heads=2,
            linear_head_dim=8,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=8,
            layer_types=["linear_attention", "full_attention"] * 2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        reference = (KlmForCausalLM(config) if causal_lm else KlmModel(config)).cuda().train()
        for name, parameter in reference.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        core = reference.model if causal_lm else reference
        core.embed_ngrams.bigram_keys.copy_(torch.arange(16, device="cuda") + 32)
        core.embed_ngrams.trigram_keys.copy_(torch.arange(8, device="cuda") + 32**2 + 32 * 5)
        core._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        (accelerated.model if causal_lm else accelerated)._use_fla_kernels = True
        if amp:
            accelerated.set_attn_implementation(f"kernels-community/flash-attn2@{FLASH_REVISION}")
        for model in (reference, accelerated):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        generator = torch.Generator().manual_seed(3407)
        batches = []
        for length in (19, 29):
            ids = torch.randint(3, 32, (2, length), generator=generator)
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[0, :3] = False
            mask[1, -4:] = False
            ids[:, 6:8] = 2
            ids[:, 12] = 2
            ids[~mask] = 0
            direction = torch.randn(2, length, 32, generator=generator)
            batches.append((ids.cuda(), mask.cuda(), direction.cuda()))
        return reference, accelerated, batches

    def _check(self, amp):
        reference, accelerated, batches = self._setup_case(amp)

        def run(model):
            losses, outputs = [], []
            for ids, mask, direction in batches:
                precision = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                with precision:
                    hidden = model(ids, attention_mask=mask, use_cache=False).last_hidden_state
                    outputs.append(hidden[mask].detach().cpu())
                    losses.append((hidden[mask].float() * direction[mask]).mean())
            (sum(losses) / len(losses)).backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                gradients[name] = parameter.grad.detach().cpu()
            return outputs, gradients

        expected_outputs, expected_gradients = run(reference)
        with patch.object(klm, "klm_chunk_gdn2", wraps=klm.klm_chunk_gdn2) as recurrence:
            actual_outputs, actual_gradients = run(accelerated)
        self.assertGreater(recurrence.call_count, 0)
        self.assertTrue(all(call.args[0].shape[-1] == 8 for call in recurrence.call_args_list))
        if amp:
            real_token_counts = {int(mask.sum()) for _, mask, _ in batches}
            self.assertTrue(all(call.args[0].shape[1] in real_token_counts for call in recurrence.call_args_list))
        self.assertEqual(actual_gradients.keys(), expected_gradients.keys())
        output_error = self._error(actual_outputs, expected_outputs)
        gradient_error = self._error(
            actual_gradients.values(), [expected_gradients[name] for name in actual_gradients]
        )
        output_limit, gradient_limit = (0.02, 0.04) if amp else (0.002, 0.01)
        result = {
            "precision": "amp" if amp else "fp32",
            "linear_head_dim": 8,
            "forward_shapes": [list(ids.shape) for ids, _, _ in batches],
            "gradient_tensors": len(actual_gradients),
            "fla_recurrence_calls": recurrence.call_count,
            "output_error": output_error,
            "gradient_error": gradient_error,
            "gates": {"output_relative_l2": output_limit, "gradient_relative_l2": gradient_limit},
        }
        SMOKE_METRICS.append(result)
        print("GPU_SMOKE_CASE " + json.dumps(result), flush=True)
        self.assertLessEqual(output_error["relative_l2"], output_limit)
        self.assertLessEqual(gradient_error["relative_l2"], gradient_limit)

    def test_small_recurrent_head_fp32(self):
        self._check(amp=False)

    @unittest.skipUnless(is_kernels_available(), "Requires the FlashAttention Hub kernel interface")
    def test_small_recurrent_head_amp_flash(self):
        self._check(amp=True)

    def _check_causal_lm(self, amp):
        for external in (False, True):
            with self.subTest(loss="external" if external else "builtin"):
                reference, accelerated, batches = self._setup_case(amp, causal_lm=True)

                def run(model):
                    losses, outputs = [], []
                    for ids, mask, _ in batches:
                        labels = ids.masked_fill(~mask, -100)
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                        with precision:
                            result = model(
                                ids, attention_mask=mask, labels=None if external else labels, use_cache=False
                            )
                            outputs.append(result.logits.detach().cpu())
                            loss = (
                                torch.nn.functional.cross_entropy(
                                    result.logits[:, :-1].float().reshape(-1, model.config.vocab_size),
                                    labels[:, 1:].reshape(-1),
                                )
                                if external
                                else result.loss
                            )
                            losses.append(loss)
                    loss = sum(losses) / len(losses)
                    loss.backward()
                    gradients = {}
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        gradients[name] = parameter.grad.detach().cpu()
                    return outputs, gradients, loss.detach().item()

                expected_outputs, expected_gradients, expected_loss = run(reference)
                with patch.object(klm, "klm_chunk_gdn2", wraps=klm.klm_chunk_gdn2) as recurrence:
                    actual_outputs, actual_gradients, actual_loss = run(accelerated)
                self.assertGreater(recurrence.call_count, 0)
                full_token_counts = {ids.numel() for ids, _, _ in batches}
                self.assertTrue(all(call.args[0].shape[1] in full_token_counts for call in recurrence.call_args_list))
                self.assertEqual(actual_gradients.keys(), expected_gradients.keys())
                output_error = self._error(actual_outputs, expected_outputs)
                gradient_error = self._error(
                    actual_gradients.values(), [expected_gradients[name] for name in actual_gradients]
                )
                output_limit, gradient_limit = (0.02, 0.04) if amp else (0.002, 0.01)
                loss_error = abs(actual_loss - expected_loss) / max(abs(expected_loss), 1e-30)
                result = {
                    "precision": "amp" if amp else "fp32",
                    "model": "causal_lm",
                    "loss_kind": "external_shifted" if external else "builtin",
                    "gradient_tensors": len(actual_gradients),
                    "fla_recurrence_calls": recurrence.call_count,
                    "padding_predictors_retained": True,
                    "output_error": output_error,
                    "gradient_error": gradient_error,
                    "loss_relative_error": loss_error,
                    "gates": {"output_relative_l2": output_limit, "gradient_relative_l2": gradient_limit},
                }
                SMOKE_METRICS.append(result)
                print("GPU_SMOKE_CASE " + json.dumps(result), flush=True)
                self.assertLessEqual(loss_error, output_limit)
                self.assertLessEqual(output_error["relative_l2"], output_limit)
                self.assertLessEqual(gradient_error["relative_l2"], gradient_limit)

    def test_causal_lm_padding_losses_fp32(self):
        self._check_causal_lm(amp=False)

    @unittest.skipUnless(is_kernels_available(), "Requires the FlashAttention Hub kernel interface")
    def test_causal_lm_padding_losses_amp_flash(self):
        self._check_causal_lm(amp=True)
